"""
DevAI MEGA Knowledge Expander
Downloads real code from GitHub, official docs, and expert Q&A content
to create a dataset comparable to professional coding assistants.

Run: python3 scripts/expand_knowledge.py
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

def fetch(url: str, label: str, timeout: int = 20) -> str:
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; DevAI/1.0)"
        })
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            content = r.read().decode("utf-8", errors="ignore")
            print(f"  ✅ {label} ({len(content)//1024} KB)")
            return content
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return ""

def clean_html(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', html)
    for ent, ch in [('&lt;','<'),('&gt;','>'),('&amp;','&'),('&quot;','"'),('&#39;',"'"),('&nbsp;',' ')]:
        text = text.replace(ent, ch)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def append(filename: str, content: str, source: str, max_bytes: int = 120000):
    path = KNOWLEDGE_DIR / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*60}\nSOURCE: {source}\n{'='*60}\n\n")
        f.write(content[:max_bytes])


# ─── PYTHON ───────────────────────────────────────────────────────────────────
def expand_python():
    print("\n📘 PYTHON — Full Official Documentation")
    pages = [
        ("https://docs.python.org/3/tutorial/index.html", "Python Tutorial"),
        ("https://docs.python.org/3/library/functions.html", "Built-in Functions"),
        ("https://docs.python.org/3/library/stdtypes.html", "Standard Types"),
        ("https://docs.python.org/3/library/exceptions.html", "Exceptions"),
        ("https://docs.python.org/3/howto/logging.html", "Logging HOWTO"),
        ("https://docs.python.org/3/library/asyncio.html", "Asyncio"),
        ("https://docs.python.org/3/library/asyncio-task.html", "Asyncio Tasks"),
        ("https://docs.python.org/3/library/concurrent.futures.html", "concurrent.futures"),
        ("https://docs.python.org/3/library/unittest.html", "Unittest"),
        ("https://docs.python.org/3/library/pathlib.html", "Pathlib"),
        ("https://docs.python.org/3/library/dataclasses.html", "Dataclasses"),
        ("https://docs.python.org/3/library/typing.html", "Type Hints"),
        ("https://docs.python.org/3/library/collections.html", "Collections"),
        ("https://docs.python.org/3/library/itertools.html", "Itertools"),
        ("https://docs.python.org/3/library/functools.html", "Functools"),
        ("https://docs.python.org/3/library/contextlib.html", "Contextlib"),
        ("https://docs.python.org/3/library/re.html", "Regex"),
        ("https://docs.python.org/3/library/json.html", "JSON"),
        ("https://docs.python.org/3/library/os.html", "OS Module"),
        ("https://docs.python.org/3/library/sys.html", "Sys Module"),
        ("https://docs.python.org/3/howto/descriptor.html", "Descriptors"),
        ("https://docs.python.org/3/reference/datamodel.html", "Data Model"),
        ("https://docs.python.org/3/library/multiprocessing.html", "Multiprocessing"),
        ("https://docs.python.org/3/library/threading.html", "Threading"),
        ("https://docs.python.org/3/library/socket.html", "Sockets"),
        ("https://docs.python.org/3/library/subprocess.html", "Subprocess"),
        ("https://docs.python.org/3/library/sqlite3.html", "SQLite3"),
        ("https://docs.python.org/3/library/csv.html", "CSV"),
        ("https://docs.python.org/3/library/enum.html", "Enum"),
        ("https://docs.python.org/3/library/abc.html", "ABC (Abstract Base Classes)"),
    ]
    for url, label in pages:
        html = fetch(url, label)
        if html:
            append("python.txt", clean_html(html), label)
        time.sleep(0.3)

    # Real-world project patterns Q&A
    append("python.txt", """
Q: How do you build a production-grade FastAPI application with authentication, database, and testing?
A:
# Project structure:
# app/
#   main.py, models.py, schemas.py, crud.py, deps.py, routers/

# models.py (SQLAlchemy)
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(String, nullable=False)
    published = Column(Boolean, default=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    author = relationship("User", back_populates="posts")

# schemas.py (Pydantic)
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

class PostCreate(BaseModel):
    title: str
    content: str
    published: bool = False

class PostResponse(PostCreate):
    id: int
    author_id: int
    class Config:
        from_attributes = True

# deps.py (Dependencies)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import User
from .auth import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    user = verify_token(token, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user

# routers/posts.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import models, schemas, deps

router = APIRouter(prefix="/posts", tags=["posts"])

@router.get("/", response_model=list[schemas.PostResponse])
def list_posts(skip: int = 0, limit: int = 20, db: Session = Depends(deps.get_db),
               current_user: models.User = Depends(deps.get_current_user)):
    return db.query(models.Post).filter(
        models.Post.author_id == current_user.id
    ).offset(skip).limit(limit).all()

@router.post("/", response_model=schemas.PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post: schemas.PostCreate, db: Session = Depends(deps.get_db),
                current_user: models.User = Depends(deps.get_current_user)):
    db_post = models.Post(**post.dict(), author_id=current_user.id)
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

# tests/test_posts.py
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..main import app
from ..database import Base
from .. import deps

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[deps.get_db] = override_get_db
client = TestClient(app)

def test_create_post():
    # First register and login
    client.post("/users/", json={"email": "test@test.com", "password": "secret"})
    login = client.post("/token", data={"username": "test@test.com", "password": "secret"})
    token = login.json()["access_token"]

    response = client.post(
        "/posts/",
        json={"title": "Test", "content": "Hello World"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Test"

Q: How do you implement async database access with SQLAlchemy 2.0?
A:
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import select

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db", echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Repository pattern
class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        result = await self.session.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, email: str, hashed_password: str) -> User:
        user = User(email=email, hashed_password=hashed_password)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

Q: How do you implement a complete Python CLI tool with rich output?
A:
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import time

app = typer.Typer(help="DevAI CLI Tool")
console = Console()

@app.command()
def list_users(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of users"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table|json")
):
    users = fetch_users(limit)
    if format == "json":
        import json
        rprint(json.dumps([u.dict() for u in users], indent=2))
    else:
        table = Table("ID", "Name", "Email", "Role", title="Users")
        for user in users:
            table.add_row(str(user.id), user.name, user.email, user.role)
        console.print(table)

@app.command()
def deploy(environment: str = typer.Argument(..., help="Target environment")):
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}")) as progress:
        task = progress.add_task("Deploying...", total=None)
        time.sleep(2)  # actual deployment
        progress.update(task, description="[green]Deployed successfully!")

if __name__ == "__main__":
    app()
""", "Python Real-World Projects Expert")


# ─── FULL STACK ────────────────────────────────────────────────────────────────
def expand_fullstack():
    print("\n🌐 FULL STACK — MDN + React + Node.js Patterns")
    pages = [
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview", "HTTP Overview"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Status", "HTTP Status Codes"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers", "HTTP Headers"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies", "HTTP Cookies"),
        ("https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch", "Fetch API"),
        ("https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API", "WebSockets"),
        ("https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API", "Web Storage"),
        ("https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API", "IndexedDB"),
        ("https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules", "JS Modules"),
        ("https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise", "Promises"),
        ("https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function", "Async Functions"),
        ("https://developer.mozilla.org/en-US/docs/Web/JavaScript/Closures", "Closures"),
        ("https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API", "Service Workers"),
        ("https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps", "PWA"),
        ("https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy", "CORS/SOP"),
    ]
    for url, label in pages:
        html = fetch(url, label)
        if html:
            append("fullstack.txt", clean_html(html), label)
        time.sleep(0.3)

    append("fullstack.txt", """
Q: How do you architect a production Next.js application with App Router?
A:
// app/layout.tsx
import { Inter } from 'next/font/google'
import { Providers } from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: { template: '%s | MyApp', default: 'MyApp' },
  description: 'Production Next.js application',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}

// app/dashboard/page.tsx
import { Suspense } from 'react'
import { DashboardSkeleton } from '@/components/skeletons'
import { getDashboardData } from '@/lib/data'

export default async function DashboardPage() {
  return (
    <main>
      <h1>Dashboard</h1>
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardContent />
      </Suspense>
    </main>
  )
}

async function DashboardContent() {
  const data = await getDashboardData() // server-side fetch, no loading state needed
  return <DashboardGrid data={data} />
}

// lib/data.ts (Server-side data fetching)
import { cache } from 'react'
import { db } from './db'

export const getUser = cache(async (id: string) => {
  return db.query.users.findFirst({ where: eq(users.id, id) })
})

// Server Action (form handling)
// app/actions.ts
'use server'
import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { z } from 'zod'

const createPostSchema = z.object({
  title: z.string().min(1).max(200),
  content: z.string().min(10),
})

export async function createPost(prevState: any, formData: FormData) {
  const validatedFields = createPostSchema.safeParse({
    title: formData.get('title'),
    content: formData.get('content'),
  })
  if (!validatedFields.success) {
    return { errors: validatedFields.error.flatten().fieldErrors }
  }
  await db.insert(posts).values(validatedFields.data)
  revalidatePath('/dashboard')
  redirect('/dashboard')
}

// Middleware (auth protection)
// middleware.ts
import { NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

export async function middleware(request) {
  const token = await getToken({ req: request })
  const isAuth = !!token
  const isAuthPage = request.nextUrl.pathname.startsWith('/auth')

  if (isAuthPage) {
    if (isAuth) return NextResponse.redirect(new URL('/dashboard', request.url))
    return null
  }
  if (!isAuth) {
    return NextResponse.redirect(new URL('/auth/login', request.url))
  }
}

export const config = { matcher: ['/dashboard/:path*', '/auth/:path*'] }

Q: How do you implement WebSocket real-time chat with Node.js?
A:
// server.js
const express = require('express')
const http = require('http')
const { Server } = require('socket.io')
const jwt = require('jsonwebtoken')

const app = express()
const server = http.createServer(app)
const io = new Server(server, {
  cors: { origin: process.env.CLIENT_URL, methods: ['GET', 'POST'] }
})

// Auth middleware for Socket.IO
io.use((socket, next) => {
  const token = socket.handshake.auth.token
  try {
    const user = jwt.verify(token, process.env.JWT_SECRET)
    socket.user = user
    next()
  } catch {
    next(new Error('Authentication failed'))
  }
})

const rooms = new Map() // roomId -> Set of userIds

io.on('connection', (socket) => {
  console.log(`User ${socket.user.id} connected`)

  socket.on('join_room', async (roomId) => {
    socket.join(roomId)
    if (!rooms.has(roomId)) rooms.set(roomId, new Set())
    rooms.get(roomId).add(socket.user.id)

    // Send room history
    const history = await getRecentMessages(roomId, 50)
    socket.emit('room_history', history)

    // Notify others
    socket.to(roomId).emit('user_joined', { userId: socket.user.id, name: socket.user.name })
  })

  socket.on('send_message', async ({ roomId, content }) => {
    const message = await saveMessage({ roomId, userId: socket.user.id, content })
    io.to(roomId).emit('new_message', {
      id: message.id,
      content,
      userId: socket.user.id,
      userName: socket.user.name,
      createdAt: message.createdAt
    })
  })

  socket.on('typing', ({ roomId }) => {
    socket.to(roomId).emit('user_typing', { userId: socket.user.id, name: socket.user.name })
  })

  socket.on('disconnect', () => {
    rooms.forEach((users, roomId) => {
      if (users.has(socket.user.id)) {
        users.delete(socket.user.id)
        socket.to(roomId).emit('user_left', { userId: socket.user.id })
      }
    })
  })
})

server.listen(3001)

Q: How do you design a microservices architecture with Docker?
A:
# docker-compose.yml — Full microservices stack
version: '3.9'
services:
  api-gateway:
    build: ./api-gateway
    ports: ["80:80"]
    depends_on: [auth-service, user-service, post-service]
    environment:
      AUTH_SERVICE_URL: http://auth-service:3001
      USER_SERVICE_URL: http://user-service:3002

  auth-service:
    build: ./services/auth
    environment:
      DATABASE_URL: postgresql://postgres:secret@auth-db:5432/auth
      JWT_SECRET: ${JWT_SECRET}
      REDIS_URL: redis://redis:6379
    depends_on: [auth-db, redis]

  user-service:
    build: ./services/user
    environment:
      DATABASE_URL: postgresql://postgres:secret@user-db:5432/users
    depends_on: [user-db]

  post-service:
    build: ./services/post
    environment:
      DATABASE_URL: postgresql://postgres:secret@post-db:5432/posts
      RABBITMQ_URL: amqp://rabbitmq
    depends_on: [post-db, rabbitmq]

  # Each service gets its own database (microservice pattern!)
  auth-db:
    image: postgres:15-alpine
    environment: { POSTGRES_DB: auth, POSTGRES_PASSWORD: secret }
    volumes: [auth-db-data:/var/lib/postgresql/data]

  user-db:
    image: postgres:15-alpine
    environment: { POSTGRES_DB: users, POSTGRES_PASSWORD: secret }

  post-db:
    image: postgres:15-alpine
    environment: { POSTGRES_DB: posts, POSTGRES_PASSWORD: secret }

  redis:
    image: redis:7-alpine

  rabbitmq:
    image: rabbitmq:3-management
    ports: ["15672:15672"] # management UI

volumes:
  auth-db-data:
""", "Full Stack Real-World Patterns")


# ─── ANDROID ──────────────────────────────────────────────────────────────────
def expand_android():
    print("\n🤖 ANDROID — Advanced Kotlin + Jetpack Compose")
    append("android.txt", """
Q: How do you implement Clean Architecture in Android with Hilt?
A:
// Domain Layer - pure Kotlin, no Android dependencies
// domain/model/User.kt
data class User(val id: String, val name: String, val email: String)

// domain/repository/UserRepository.kt (interface)
interface UserRepository {
    fun getUsers(): Flow<List<User>>
    suspend fun getUserById(id: String): User?
    suspend fun updateUser(user: User): Result<User>
}

// domain/usecase/GetUsersUseCase.kt
class GetUsersUseCase(private val repository: UserRepository) {
    operator fun invoke(): Flow<List<User>> = repository.getUsers()
        .map { users -> users.sortedBy { it.name } }
        .catch { e -> throw DomainException("Failed to get users", e) }
}

// Data Layer
// data/remote/UserApi.kt
interface UserApi {
    @GET("users")
    suspend fun getUsers(): List<UserDto>
    @GET("users/{id}")
    suspend fun getUser(@Path("id") id: String): UserDto
}

// data/repository/UserRepositoryImpl.kt
class UserRepositoryImpl(
    private val api: UserApi,
    private val dao: UserDao,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) : UserRepository {
    override fun getUsers(): Flow<List<User>> = flow {
        // Cache-first strategy
        emit(dao.getAll().map { it.toDomain() })
        // Refresh from network
        val fresh = withContext(dispatcher) { api.getUsers() }
        dao.upsertAll(fresh.map { it.toEntity() })
        emit(fresh.map { it.toDomain() })
    }.flowOn(dispatcher)
}

// Presentation Layer
// presentation/users/UsersViewModel.kt
@HiltViewModel
class UsersViewModel @Inject constructor(
    private val getUsersUseCase: GetUsersUseCase
) : ViewModel() {

    sealed class UiState {
        object Loading : UiState()
        data class Success(val users: List<User>) : UiState()
        data class Error(val message: String) : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.Loading)
    val uiState = _uiState.asStateFlow()

    // Search support
    private val _searchQuery = MutableStateFlow("")
    val searchQuery = _searchQuery.asStateFlow()

    val filteredUsers = combine(uiState, searchQuery) { state, query ->
        if (state is UiState.Success && query.isNotBlank()) {
            UiState.Success(state.users.filter { it.name.contains(query, ignoreCase = true) })
        } else state
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), UiState.Loading)

    init {
        loadUsers()
    }

    private fun loadUsers() {
        getUsersUseCase()
            .onStart { _uiState.value = UiState.Loading }
            .onEach { _uiState.value = UiState.Success(it) }
            .catch { _uiState.value = UiState.Error(it.message ?: "Unknown error") }
            .launchIn(viewModelScope)
    }

    fun onSearchQueryChange(query: String) { _searchQuery.value = query }
}

Q: How do you implement WorkManager for background tasks in Android?
A:
// Define work
class SyncWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        return try {
            val userId = inputData.getString("userId") ?: return Result.failure()
            // Do actual work
            syncRepository.syncUser(userId)
            Result.success()
        } catch (e: Exception) {
            if (runAttemptCount < 3) Result.retry() else Result.failure()
        }
    }

    companion object {
        fun buildRequest(userId: String): OneTimeWorkRequest {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()
            return OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(constraints)
                .setInputData(workDataOf("userId" to userId))
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
                .addTag("sync")
                .build()
        }
    }
}

// Schedule work
class UserRepository(private val workManager: WorkManager) {
    fun scheduleSync(userId: String) {
        val request = SyncWorker.buildRequest(userId)
        workManager.enqueueUniqueWork("user-sync-$userId",
            ExistingWorkPolicy.REPLACE, request)
    }

    fun observeSyncStatus(userId: String): Flow<WorkInfo?> =
        workManager.getWorkInfosByTagFlow("sync").map { it.firstOrNull() }
}

Q: How do you handle offline-first in Android with Room + Retrofit?
A:
// NetworkBoundResource pattern
inline fun <ResultType, RequestType> networkBoundResource(
    crossinline query: () -> Flow<ResultType>,
    crossinline fetch: suspend () -> RequestType,
    crossinline saveFetchResult: suspend (RequestType) -> Unit,
    crossinline shouldFetch: (ResultType) -> Boolean = { true }
) = flow {
    val data = query().first()
    val flow = if (shouldFetch(data)) {
        emit(Resource.Loading(data))
        try {
            saveFetchResult(fetch())
            query().map { Resource.Success(it) }
        } catch (t: Throwable) {
            query().map { Resource.Error(t, it) }
        }
    } else {
        query().map { Resource.Success(it) }
    }
    emitAll(flow)
}

// Usage
fun getArticles(): Flow<Resource<List<Article>>> = networkBoundResource(
    query = { dao.getArticles() },
    fetch = { api.getArticles() },
    saveFetchResult = { articles -> dao.upsert(articles.map { it.toEntity() }) },
    shouldFetch = { cached -> cached.isEmpty() || isStale() }
)
""", "Android Clean Architecture Expert")


# ─── iOS ──────────────────────────────────────────────────────────────────────
def expand_ios():
    print("\n🍎 iOS — SwiftUI + Combine + Swift Concurrency")
    append("ios.txt", """
Q: How do you build a production iOS app with SwiftUI and TCA (The Composable Architecture)?
A:
// Feature.swift
import ComposableArchitecture

struct UserFeature: Reducer {
    struct State: Equatable {
        var users: [User] = []
        var isLoading = false
        var searchText = ""
        var selectedUser: User?
        var alert: AlertState<Action>?

        var filteredUsers: [User] {
            searchText.isEmpty ? users : users.filter {
                $0.name.localizedCaseInsensitiveContains(searchText)
            }
        }
    }

    enum Action: Equatable {
        case loadUsers
        case usersLoaded(TaskResult<[User]>)
        case searchTextChanged(String)
        case userTapped(User)
        case alertDismissed
    }

    @Dependency(\\.userClient) var userClient

    var body: some ReducerOf<Self> {
        Reduce { state, action in
            switch action {
            case .loadUsers:
                state.isLoading = true
                return .run { send in
                    await send(.usersLoaded(TaskResult { try await userClient.fetchAll() }))
                }
            case let .usersLoaded(.success(users)):
                state.isLoading = false
                state.users = users
                return .none
            case let .usersLoaded(.failure(error)):
                state.isLoading = false
                state.alert = AlertState { TextState("Error: \\(error.localizedDescription)") }
                return .none
            case let .searchTextChanged(text):
                state.searchText = text
                return .none
            case let .userTapped(user):
                state.selectedUser = user
                return .none
            case .alertDismissed:
                state.alert = nil
                return .none
            }
        }
    }
}

Q: How do you implement advanced networking in Swift with async/await?
A:
// NetworkClient.swift
actor NetworkClient {
    private let session: URLSession
    private let decoder: JSONDecoder
    private let baseURL: URL
    private var authToken: String?

    init(baseURL: URL) {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        session = URLSession(configuration: config)
        decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        self.baseURL = baseURL
    }

    func setAuthToken(_ token: String) { authToken = token }

    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        var request = URLRequest(url: baseURL.appendingPathComponent(endpoint.path))
        request.httpMethod = endpoint.method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token = authToken {
            request.setValue("Bearer \\(token)", forHTTPHeaderField: "Authorization")
        }
        if let body = endpoint.body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }

        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        switch http.statusCode {
        case 200..<300: break
        case 401: throw NetworkError.unauthorized
        case 404: throw NetworkError.notFound
        case 500..<600: throw NetworkError.serverError(http.statusCode)
        default: throw NetworkError.unknown(http.statusCode)
        }

        return try decoder.decode(T.self, from: data)
    }
}

Q: How do you use SwiftData for local persistence in iOS 17+?
A:
import SwiftData

@Model
class Task {
    @Attribute(.unique) var id: UUID
    var title: String
    var isCompleted: Bool
    var priority: Priority
    var createdAt: Date
    var tags: [String]

    @Relationship(deleteRule: .cascade)
    var subtasks: [SubTask]?

    init(title: String, priority: Priority = .medium) {
        self.id = UUID()
        self.title = title
        self.isCompleted = false
        self.priority = priority
        self.createdAt = Date()
        self.tags = []
    }

    enum Priority: Int, Codable {
        case low = 1, medium = 2, high = 3
    }
}

// In App
@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: Task.self)
    }
}

// In View
struct TaskListView: View {
    @Query(sort: \\\\Task.createdAt, order: .reverse) private var tasks: [Task]
    @Environment(\\\\.modelContext) private var context

    var body: some View {
        List(tasks) { task in
            TaskRow(task: task)
        }
        .toolbar {
            Button(action: addTask) {
                Label("Add", systemImage: "plus")
            }
        }
    }

    func addTask() {
        let task = Task(title: 'New Task')
        context.insert(task)
    }
}
""", "iOS Advanced Patterns Expert")


# ─── JAVA ─────────────────────────────────────────────────────────────────────
def expand_java():
    print("\n☕ JAVA — Spring Boot + Microservices")
    append("java.txt", """
Q: How do you build a complete Spring Boot microservice with all best practices?
A:
// Entity
@Entity
@Table(name = "orders")
@EntityListeners(AuditingEntityListener.class)
@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class Order {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String customerId;

    @Enumerated(EnumType.STRING)
    private OrderStatus status;

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderItem> items = new ArrayList<>();

    @CreatedDate private LocalDateTime createdAt;
    @LastModifiedDate private LocalDateTime updatedAt;

    public BigDecimal getTotal() {
        return items.stream()
            .map(item -> item.getPrice().multiply(new BigDecimal(item.getQuantity())))
            .reduce(BigDecimal.ZERO, BigDecimal::add);
    }
}

// Service
@Service
@Transactional
@RequiredArgsConstructor
@Slf4j
public class OrderService {
    private final OrderRepository orderRepository;
    private final ProductClient productClient; // Feign client
    private final KafkaTemplate<String, OrderEvent> kafka;
    private final OrderMapper mapper;

    public OrderResponse createOrder(CreateOrderRequest request) {
        // Validate inventory via product service
        request.getItems().forEach(item -> {
            ProductResponse product = productClient.getProduct(item.getProductId());
            if (product.getStock() < item.getQuantity()) {
                throw new InsufficientStockException(item.getProductId());
            }
        });

        Order order = mapper.toEntity(request);
        order.setStatus(OrderStatus.PENDING);
        Order saved = orderRepository.save(order);

        // Publish domain event
        kafka.send("orders", saved.getId().toString(),
            new OrderCreatedEvent(saved.getId(), saved.getCustomerId(), saved.getTotal()));

        log.info("Order {} created for customer {}", saved.getId(), saved.getCustomerId());
        return mapper.toResponse(saved);
    }
}

// Global Exception Handler
@RestControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ProblemDetail handleValidation(MethodArgumentNotValidException ex) {
        ProblemDetail detail = ProblemDetail.forStatus(HttpStatus.BAD_REQUEST);
        detail.setTitle("Validation Failed");
        detail.setProperty("errors", ex.getBindingResult().getFieldErrors().stream()
            .collect(Collectors.toMap(
                FieldError::getField,
                fe -> fe.getDefaultMessage() != null ? fe.getDefaultMessage() : "Invalid"
            )));
        return detail;
    }

    @ExceptionHandler(ResourceNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ProblemDetail handleNotFound(ResourceNotFoundException ex) {
        ProblemDetail detail = ProblemDetail.forStatus(HttpStatus.NOT_FOUND);
        detail.setTitle(ex.getMessage());
        return detail;
    }
}

// Integration Test
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestContainers
class OrderServiceIntegrationTest {
    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15");

    @Container
    static KafkaContainer kafka = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka"));

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.kafka.bootstrap-servers", kafka::getBootstrapServers);
    }

    @Autowired private MockMvc mockMvc;

    @Test
    void createOrder_success() throws Exception {
        mockMvc.perform(post("/api/v1/orders")
            .contentType(MediaType.APPLICATION_JSON)
            .content("{\\"customerId\\": \\"cust-1\\", \\"items\\": [{\\"productId\\": \\"prod-1\\", \\"quantity\\": 2}]}")
            .andExpect(status().isCreated())
            .andExpect(jsonPath("$.status").value("PENDING"))
            .andExpect(jsonPath("$.customerId").value("cust-1"));
    }
}
""", "Java Spring Boot Expert")


# ─── NETWORKS ─────────────────────────────────────────────────────────────────
def expand_networks():
    print("\n🌍 NETWORKS — Deep Technical Content")
    pages = [
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview", "HTTP Overview"),
        ("https://developer.mozilla.org/en-US/docs/Web/Security/Transport_Layer_Security", "TLS"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication", "HTTP Auth"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching", "HTTP Caching"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS", "CORS"),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP", "CSP"),
        ("https://developer.mozilla.org/en-US/docs/Glossary/TCP", "TCP"),
        ("https://developer.mozilla.org/en-US/docs/Glossary/UDP", "UDP"),
        ("https://developer.mozilla.org/en-US/docs/Glossary/DNS", "DNS"),
    ]
    for url, label in pages:
        html = fetch(url, label)
        if html:
            append("networks.txt", clean_html(html), label)
        time.sleep(0.3)

    append("networks.txt", """
Q: Explain the full HTTP/2 and HTTP/3 improvements over HTTP/1.1?
A:
HTTP/1.1 Problems:
- Head-of-line blocking: requests queued on same TCP connection
- No header compression: redundant headers sent every request
- Only text protocol: inefficient binary parsing

HTTP/2 Solutions (2015):
- Multiplexing: multiple streams on ONE TCP connection (no head-of-line blocking at HTTP layer)
- Header compression (HPACK): reference tables for repeated headers
- Server push: server can proactively send resources
- Binary protocol: frames instead of text
- Stream prioritization

HTTP/3 Solutions (2022):
- Runs on QUIC protocol (built on UDP) instead of TCP
- Eliminates TCP head-of-line blocking entirely (each stream independent)
- Built-in TLS 1.3 (0-RTT connection resumption for returning clients)
- Connection migration: switch networks (WiFi→4G) without dropping connection
- Faster connection setup: 0-RTT vs 3-way handshake + TLS handshake

Q: How do you implement rate limiting in an API gateway?
A:
// Token Bucket Algorithm in Redis
class RateLimiter:
    def __init__(self, redis_client, max_requests=100, window_seconds=60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds

    def is_allowed(self, identifier: str) -> tuple[bool, dict]:
        key = f"rate:{identifier}"
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - self.window

        pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
        pipe.zadd(key, {str(now): now})              # Add current request
        pipe.zcard(key)                               # Count requests
        pipe.expire(key, self.window)                 # Set expiry
        results = pipe.execute()

        count = results[2]
        remaining = max(0, self.max_requests - count)
        reset_at = now + self.window

        return count <= self.max_requests, {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_at)),
        }

Q: How does a CDN work and how should you configure caching headers?
A:
CDN (Content Delivery Network) flow:
1. User requests resource from edge node (closest geographically)
2. Edge node checks local cache: HIT → return immediately
3. Cache MISS → edge fetches from origin server
4. Edge caches response based on Cache-Control headers
5. Future requests for same resource served from edge

Cache-Control headers:
Cache-Control: public, max-age=31536000, immutable
  - public: can be cached by CDN and browsers
  - max-age=31536000: cache for 1 year (use with content hashing!)
  - immutable: browser won't revalidate even on reload

Cache-Control: private, no-cache
  - private: only browser cache, NOT CDN
  - no-cache: must revalidate with server before serving

Cache-Control: no-store
  - Never cache (sensitive data: banking, auth tokens)

ETag pattern (conditional requests):
Server sends: ETag: "abc123"
Browser sends on next request: If-None-Match: "abc123"
Server returns 304 Not Modified if unchanged (no body = less bandwidth)
""", "Networks Deep Dive Expert")


# ─── DATABASES ────────────────────────────────────────────────────────────────
def expand_databases():
    print("\n🗄️  DATABASES — SQL + NoSQL + Performance")
    pages = [
        ("https://www.postgresql.org/docs/current/sql-select.html", "PostgreSQL SELECT"),
        ("https://www.sqlite.org/index.html", "SQLite Home"),
    ]
    for url, label in pages:
        html = fetch(url, label)
        if html:
            append("dbms.txt", clean_html(html), label)
            append("sql.txt", clean_html(html), label)
        time.sleep(0.3)

    deep_content = """
Q: How do you optimize slow PostgreSQL queries?
A:
STEP 1: Find slow queries
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;

STEP 2: EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.*, COUNT(o.id) as order_count
FROM users u LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id;

Key terms to look for:
- Seq Scan (bad for large tables) → needs an index
- Index Scan (good) → using index
- Hash Join vs Nested Loop vs Merge Join
- Actual vs Estimated rows: big difference = outdated statistics
  → Fix: ANALYZE table_name

STEP 3: Add indexes strategically
-- Index on commonly filtered column
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
-- For joins
CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);
-- Composite for multi-column WHERE
CREATE INDEX CONCURRENTLY idx_orders_user_status ON orders(user_id, status);
-- Partial index (only active records)
CREATE INDEX idx_active_orders ON orders(user_id) WHERE status = 'active';

STEP 4: Query rewrites
-- Bad: function on indexed column (disables index!)
WHERE LOWER(email) = 'test@example.com'
-- Good: expression index
CREATE INDEX idx_users_email_lower ON users(LOWER(email));
WHERE LOWER(email) = 'test@example.com' -- now uses index!

-- Bad: SELECT * (fetches all columns)
-- Good: SELECT only needed columns (enables index-only scans)
SELECT id, name FROM users WHERE id = 1;

STEP 5: Connection pooling (PgBouncer)
Postgres creates a process per connection (expensive!)
PgBouncer pools connections: app has 1000 "connections", DB sees 20.

Q: When should you use NoSQL vs SQL?
A:
Use PostgreSQL (SQL) when:
- Data has clear relationships (users, orders, products)
- You need ACID transactions (financial, inventory)
- You need complex queries with JOINs and aggregations
- Schema is known upfront and relatively stable

Use MongoDB (Document) when:
- Data structure varies between records
- Working with hierarchical/nested data (no need to join)
- Rapid prototyping (flexible schema)
- Example: CMS, catalogs, user profiles with arbitrary fields

Use Redis (Key-Value) when:
- Caching database query results
- Session storage
- Rate limiting counters
- Real-time leaderboards (sorted sets)
- Pub/Sub messaging

Use Cassandra (Wide Column) when:
- Time-series data (metrics, events, logs)
- Write-heavy workloads at extreme scale
- Data is naturally partitioned by a key (sensor_id, user_id)

Use Elasticsearch (Search) when:
- Full-text search with relevance scoring
- Log analytics (ELK stack)
- Faceted filtering (e-commerce product search)

Q: How do you implement database migrations properly?
A:
Using Alembic (Python):
# alembic/versions/001_create_users.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table('users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(100)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('idx_users_email', 'users', ['email'])

def downgrade():
    op.drop_table('users')

# Adding column safely (zero-downtime)
def upgrade():
    # Step 1: Add nullable column (instant)
    op.add_column('orders', sa.Column('tracking_number', sa.String(50), nullable=True))
    # Step 2: Backfill in batches (do NOT do UPDATE orders SET tracking_number = '' -- locks table!)
    # Step 3: Make NOT NULL after backfill (separate migration)

# Zero-downtime index creation
def upgrade():
    op.execute('CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status)')
    # CONCURRENTLY builds index without locking writes!
"""
    append("dbms.txt", deep_content, "Database Performance Expert")
    append("sql.txt", deep_content, "SQL Performance Expert")
    append("sqlite.txt", """
Q: How do you use SQLite effectively in production apps?
A:
SQLite is NOT just for development! WhatsApp, Firefox, iOS, Android all use SQLite.

Best practices:
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db(path: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Access columns by name
    conn.execute("PRAGMA journal_mode=WAL")   # WAL mode: concurrent reads+writes
    conn.execute("PRAGMA synchronous=NORMAL") # Faster, still safe
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in RAM
    conn.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

# Full-text search with FTS5
CREATE VIRTUAL TABLE posts_fts USING fts5(title, content, content=posts, content_rowid=id);
CREATE TRIGGER posts_ai AFTER INSERT ON posts BEGIN
    INSERT INTO posts_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;
SELECT * FROM posts JOIN posts_fts ON posts.id = posts_fts.rowid
WHERE posts_fts MATCH 'python AND async';
""", "SQLite Expert")


# ─── OS ───────────────────────────────────────────────────────────────────────
def expand_os():
    print("\n💻 OPERATING SYSTEMS — Deep Technical Content")
    append("os.txt", """
Q: How does virtual memory work at a hardware level?
A:
Virtual Memory Components:
1. MMU (Memory Management Unit) — hardware unit in CPU
2. TLB (Translation Lookaside Buffer) — cache of recent virtual→physical mappings
3. Page Table — data structure mapping virtual pages to physical frames

Address Translation:
Virtual address: [VPN | Page Offset]  (e.g., 48-bit: 36-bit VPN + 12-bit offset)
Physical address: [PFN | Page Offset]

Process:
1. CPU generates virtual address
2. MMU looks up VPN in TLB (if HIT → fast, get PFN)
3. TLB MISS → walk page table in memory (slow, 4+ memory accesses)
4. On TLB miss: x86 hardware walks the 4-level page table automatically
5. Page Fault: page not in RAM → OS loads from disk → update page table → retry

Copy-on-Write (COW) — How fork() is fast:
After fork(), parent and child share same physical pages (mapped read-only)
When either writes → page fault → OS copies page → both get their own copy
99% of forked processes exec() immediately → no copying needed!

Q: How does the Linux kernel scheduler work?
A:
CFS (Completely Fair Scheduler) — Linux since 2.6.23:

Key concept: vruntime (virtual runtime)
- Each task accumulates vruntime as it runs
- Task with SMALLEST vruntime runs next
- Nice value scales vruntime accumulation (lower priority = faster accumulation)

Data structure: Red-Black tree (self-balancing BST)
- O(log n) insert/delete
- Leftmost node = smallest vruntime = next to run

Time slice: Not fixed! Calculated as:
timeslice = (sched_latency / n_tasks) * (weight / total_weight)
sched_latency default = 6ms (every task runs at least once per 6ms)

Multicore scheduling:
- Per-CPU run queues (no global lock = scalable)
- Load balancing via work stealing (idle CPU pulls tasks from busy CPUs)
- NUMA awareness: prefer to schedule on same NUMA node as task's memory

Q: What happens when you type a command in the terminal?
A:
1. Shell (bash/zsh) reads input: ls -la /home
2. Shell forks() a child process (vfork on Linux for efficiency)
3. Child calls execve("/bin/ls", ["ls", "-la", "/home"], env)
   - execve replaces child's memory with ls binary
   - Loads ELF binary headers
   - Dynamic linker (ld.so) resolves shared library symbols
4. Shell waits via waitpid() or puts in background
5. ls opens /home via openat() syscall → VFS → ext4 filesystem
6. ls reads directory entries via getdents64() syscall
7. ls writes output to stdout (fd=1) via write() syscall
8. Terminal emulator receives output bytes via pty master
9. ls exits → kernel sends SIGCHLD to shell → shell resumes
""", "OS Deep Dive Expert")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("🚀 DevAI MEGA Knowledge Expander")
    print("="*60)

    expand_python()
    expand_fullstack()
    expand_android()
    expand_ios()
    expand_java()
    expand_networks()
    expand_databases()
    expand_os()

    print("\n📊 Final Knowledge Base:")
    total = 0
    for f in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        size = f.stat().st_size
        total += size
        bar = "█" * min(40, size // 25000)
        print(f"  {f.name:20s} {size//1024:>5} KB  {bar}")
    print(f"\n  {'TOTAL':20s} {total//1024:>5} KB  ({total//(1024*1024)} MB)")
    print("\n✅ Done! Now run: python3 scripts/build_corpus.py")

if __name__ == "__main__":
    main()
