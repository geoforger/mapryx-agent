# Mapryx AI Agent Context

This document provides complete context for AI agents working with the Mapryx codebase. The agent is tightly coupled with the main Mapryx repository located at `/Users/abdulrehman/products/mapryx2`.

## Repository Structure

```
/Users/abdulrehman/products/mapryx2/
├── src/                          # Main Mapryx source code
│   ├── ai/                       # AI Assistant core
│   │   ├── executor.ts           # Tool execution logic
│   │   ├── orchestrator.ts       # System prompt generation
│   │   └── llm.ts                # LLM API integration
│   ├── core/                     # DuckDB-WASM integration
│   │   ├── duckdb.ts             # Database connection & queries
│   │   ├── crsQuery.ts           # Coordinate reference system queries
│   │   ├── MVTExtractor.ts       # MVT tile extraction
│   │   ├── BundleService.ts      # Bundle management
│   │   ├── ExportService.ts      # Export functionality
│   │   ├── crsService.ts         # CRS handling
│   │   ├── EditManager.ts        # Edit operations
│   │   └── analytics/            # Analytics services
│   ├── loaders/                  # File format loaders
│   │   ├── geoparquet.ts         # GeoParquet loading
│   │   ├── geojson.ts            # GeoJSON loading
│   │   ├── geometry.ts           # WKB geometry parsing
│   │   ├── csv.ts                # CSV loading
│   │   ├── shapefile.ts          # Shapefile loading
│   │   ├── kml.ts                # KML loading
│   │   ├── gpx.ts                # GPX loading
│   │   ├── googleSheets.ts       # Google Sheets loading
│   │   ├── paste.ts              # Pasted data handling
│   │   ├── plainTable.ts         # Plain table loading
│   │   ├── geopackage.ts         # GeoPackage loading
│   │   ├── spatialite.ts         # SpatiaLite loading
│   │   └── duckdbTable.ts        # DuckDB table operations
│   ├── stores/                   # Zustand state stores
│   │   ├── layerStore.ts         # Layer management
│   │   ├── uiStore.ts            # UI state
│   │   ├── settingsStore.ts      # User settings
│   │   ├── authStore.ts          # Authentication
│   │   ├── feedbackStore.ts      # Feedback system
│   │   ├── printStore.ts         # Print functionality
│   │   ├── analyticsStore.ts     # Analytics state
│   │   └── searchStore.ts        # Search state
│   ├── layers/                   # deck.gl layer factories
│   │   ├── factory.ts            # Layer factory
│   │   └── labels.ts             # Label layers
│   ├── components/               # React components
│   │   ├── panels/               # Panel components
│   │   ├── modals/               # Modal dialogs
│   │   ├── drawer/               # Drawer components
│   │   ├── ui/                   # UI primitives
│   │   ├── table/                # Table components
│   │   ├── labels/               # Label components
│   │   ├── search/               # Search components
│   │   ├── print/                # Print components
│   │   ├── feedback/             # Feedback components
│   │   ├── analytics/            # Analytics components
│   │   ├── auth/                 # Auth components
│   │   ├── map/                  # Map controls
│   │   └── ...                   # Other components
│   ├── hooks/                    # Custom React hooks
│   │   ├── useMapryxAI.ts        # AI assistant hook
│   │   ├── useMapryxChatAgent.ts # Agent chat hook
│   │   ├── useDeckLayers.ts      # Deck.gl layer hook
│   │   ├── useLayerLabels.ts     # Layer labels hook
│   │   └── ...                   # Other hooks
│   ├── utils/                    # Utility functions
│   │   ├── ai-context.ts         # AI context generation
│   │   ├── ai-ingestion.ts       # AI result provisioning
│   │   ├── layerContext.ts       # Layer context management
│   │   ├── parameterValidation.ts # Parameter validation
│   │   ├── colorUtils.ts         # Color parsing utilities
│   │   ├── layerIdentification.ts # Layer identification
│   │   ├── filterUtils.ts        # Filter utilities
│   │   ├── format.ts             # Formatting utilities
│   │   ├── storage.ts            # Storage utilities
│   │   ├── network.ts            # Network utilities
│   │   └── ...                   # Other utilities
│   ├── types/                    # TypeScript definitions
│   │   └── index.ts              # Main type definitions
│   ├── constants/                # Constants
│   │   └── print.ts              # Print constants
│   ├── lib/                      # Library code
│   │   ├── duckSpatialEngine.ts  # DuckDB spatial engine
│   │   └── supabase.ts           # Supabase integration
│   ├── services/                 # Backend services
│   │   └── feedbackService.ts    # Feedback service
│   ├── tools/                    # Tool definitions
│   │   └── registry.ts           # Tool registry
│   ├── workers/                  # Web workers
│   │   ├── plain-table-worker.ts # Plain table worker
│   │   └── geometry-worker.ts    # Geometry worker
│   ├── shims/                    # TypeScript shims
│   │   └── child-process.ts      # Child process shim
│   ├── pages/                    # Page components
│   │   ├── MapPage.tsx           # Main map page
│   │   └── PrintPage.tsx         # Print page
│   ├── assets/                   # Static assets
│   └── main.tsx                  # Application entry point
├── .agents/                      # AI agent configuration
│   ├── rules/                    # Agent rules
│   │   └── coding-guide.md       # Coding standards
│   ├── skills/                   # Agent skills
│   │   ├── MapryxCore/           # Core Mapryx skill
│   │   │   ├── SKILL.md          # Skill documentation
│   │   │   ├── scripts/          # Skill scripts
│   │   │   └── examples/         # Skill examples
│   │   └── GitCommitFlow/        # Git workflow skill
│   └── workflows/                # Agent workflows
│       ├── git-commit.md         # Commit workflow
│       ├── git-pr.md             # PR workflow
│       ├── github-issues.md      # Issue workflow
│       └── release-process.md    # Release workflow
├── .kiro/                        # Kiro agent settings
│   ├── specs/                    # Feature specifications
│   ├── steering/                 # Steering rules
│   │   ├── documentation-organization.md
│   │   ├── no-root-markdown-files.md
│   │   └── testing-requirements.md
│   ├── hooks/                    # Agent hooks
│   │   └── commit-check-errors.kiro.hook
│   └── settings/                 # Agent settings
├── docs/                         # Documentation
│   ├── ai-assistant/             # AI assistant docs
│   ├── features/                 # Feature docs
│   ├── implementation/           # Implementation guides
│   └── screenshots/              # Screenshots
├── tests/                        # Test files
├── playwright-report/            # Playwright reports
├── test-results/                 # Test results
├── scripts/                      # Build scripts
├── public/                       # Public assets
├── dist/                         # Build output
├── package.json                  # Dependencies
├── tsconfig.json                 # TypeScript config
├── vite.config.ts                # Vite configuration
├── eslint.config.js              # ESLint configuration
├── vitest.setup.ts               # Vitest setup
├── playwright.config.ts          # Playwright config
└── README.md                     # Project README
```

## Key AI Agent Files

### 1. `/Users/abdulrehman/products/mapryx2/src/ai/executor.ts`

**Purpose**: Executes all spatial tools and queries called by the AI assistant.

**Key Functions**:
- `executeSpatialQuery(query, description)` - Execute DuckDB spatial SQL
- `executeSelectLayer(layerName, filter, outputName)` - Filter layer
- `executeBufferLayer(layerName, distance, units, outputName)` - Buffer layer
- `executeClipLayer(layerName, maskLayerName, outputName)` - Clip layer
- `executeIntersectLayers(layerName, targetLayerName, outputName)` - Intersect layers
- `executeDissolveLayer(layerName, groupByAttribute, outputName)` - Dissolve layer
- `executeSplitLayer(layerName, splitByAttribute, outputName)` - Split layer
- `executeUnionLayer(layerName, outputName)` - Union layer
- `executeCategorizeLayer(layerName, attributeName)` - Categorize layer
- `executeFilterLayer(layerName, filters)` - Filter layer
- `executeToggleCategoryVisibility(layerName, categoryValue, action)` - Toggle visibility
- `executeUpdateLayerStyle(layerName, style)` - Update layer style
- `executeListLayers(includeAttributes, visibleOnly)` - List layers
- `executeCheckSystemStatus()` - Check system status
- `executeQuerySpatialFunctions(functionPattern)` - Query spatial functions
- `executeAttributeJoinLayers(...)` - Attribute join
- `executeSpatialJoinLayers(...)` - Spatial join
- `executeCalculateGeometry(...)` - Calculate geometry
- `executeCalculateField(...)` - Calculate field
- `executeInspectLayer(layerName)` - Inspect layer
- `executeGetUniqueValues(layerName, attributeName, limit)` - Get unique values

### 2. `/Users/abdulrehman/products/mapryx2/src/ai/orchestrator.ts`

**Purpose**: Generates system prompts and defines available tools.

**Key Functions**:
- `generateSystemPrompt(query)` - Generate system prompt with context
- `SPATIAL_TOOLS` - Array of available assistant tools

**System Prompt Includes**:
- Table name guard rules (use internal `tableName`, not user-facing names)
- Tool routing guidance
- DuckDB-WASM spatial SQL syntax
- Coordinate system rules (use strings like `'EPSG:4326'`, not integers)
- Available spatial functions
- Schema context from current database

### 3. `/Users/abdulrehman/products/mapryx2/src/ai/llm.ts`

**Purpose**: Handles LLM API calls (OpenAI-compatible and Google AI).

**Key Functions**:
- `callLLM(options)` - Call LLM with messages and tools
- Supports Google AI native format
- Supports OpenAI-compatible endpoints
- Handles mixed content/CORS issues for localhost endpoints

### 4. `/Users/abdulrehman/products/mapryx2/src/hooks/useMapryxAI.ts`

**Purpose**: React hook for AI assistant chat interface.

**Key Features**:
- Message history management
- Tool call execution loop
- Error handling with guidance
- Maximum iteration limits
- Consecutive error tracking

### 5. `/Users/abdulrehman/products/mapryx2/src/hooks/useMapryxChatAgent.ts`

**Purpose**: React hook for Cloudflare Agents integration.

**Key Features**:
- Session token management
- Agent connection with authentication
- Tool call handling
- Streaming support

### 6. `/Users/abdulrehman/products/mapryx2/src/utils/ai-context.ts`

**Purpose**: Generates database context for AI assistant.

**Key Functions**:
- `getDatabaseContext(query)` - Get optimized database context
- `createConfiguredContext(config)` - Create configured context builder
- `getLayerTableMappingContext()` - Get layer-to-table mapping

### 7. `/Users/abdulrehman/products/mapryx2/src/utils/ai-ingestion.ts`

**Purpose**: Provisions AI-generated results for map display.

**Key Functions**:
- `provisionAIResult(tableName, sourceQuery, geometryCol, isExistingTable)` - Provision AI result with:
  - Geometry cleanup (ST_MakeValid, ST_Multi, ST_CollectionExtract)
  - R-Tree index creation
  - Statistics update

### 8. `/Users/abdulrehman/products/mapryx2/src/tools/registry.ts`

**Purpose**: Defines all available spatial tools.

**Key Types**:
- `SpatialToolId` - Tool identifiers
- `ToolCategory` - Tool categories
- `SpatialToolDefinition` - Tool definitions
- `AssistantToolDefinition` - Assistant tool definitions
- `ASSISTANT_TOOLS` - Array of 24 available tools

### 9. `/Users/abdulrehman/products/mapryx2/src/types/index.ts`

**Purpose**: Core TypeScript type definitions.

**Key Types**:
- `GeometryType` - Point, MultiPoint, LineString, etc.
- `ServiceLayerType` - MVT, WMS, WFS, ESRI_FEATURE, etc.
- `LayerConfig` - Layer configuration
- `LayerData` - Layer data for rendering
- `LayerStyle` - Layer styling
- `LayerFilters` - Layer filters
- `ChatMessage` - Chat message structure
- `ChatSession` - Chat session structure
- `UIState` - UI state
- `LayerStoreState` - Layer store state

## Core Architecture

### Data Flow
```
User Input → LLM → Tool Calls → Executor → DuckDB-WASM → Results → UI
```

### Layer Management
- Layers stored in `useLayerStore` (Zustand)
- Each layer has: `id`, `name`, `visible`, `data`, `style`, `tableName`
- Data stored in DuckDB-WASM with internal table names
- Layers rendered via deck.gl with binary geometry format

### Spatial SQL Rules
1. **Always use internal table names** (e.g., `table_geojson_123`)
2. **Never use user-facing layer names** in SQL queries
3. **Always alias geometry as `geom`**
4. **Use STRING SRIDs** (e.g., `'EPSG:4326'`, not `4326`)
5. **ST_Transform requires 3 arguments**: `ST_Transform(geom, 'source', 'target')`

### Available Tools (24 total)
1. `run_spatial_query` - Execute DuckDB SQL
2. `select_layer` - Filter layer
3. `buffer_layer` - Buffer features
4. `clip_layer` - Clip by mask
5. `intersect_layers` - Intersect two layers
6. `dissolve_layer` - Dissolve by attribute
7. `split_layer` - Split by attribute
8. `union_layer` - Union all geometries
9. `check_system_status` - Check system
10. `query_spatial_functions` - Query functions
11. `categorize_layer` - Categorize symbology
12. `filter_layer` - Attribute filters
13. `toggle_category_visibility` - Toggle categories
14. `update_layer_style` - Update styling
15. `list_layers` - List layers
16. `attribute_join_layers` - Attribute join
17. `spatial_join_layers` - Spatial join
18. `calculate_geometry` - Calculate area/length
19. `calculate_field` - Calculate custom field
20. `inspect_layer` - Inspect layer schema
21. `get_unique_values` - Get unique values
22. `extract_mvt_data` - Extract MVT data
23. `points_to_path` - Points to path
24. `merge_layers` - Merge layers
25. `erase_layer` - Erase features

## Development Setup

### Main Project
```bash
cd /Users/abdulrehman/products/mapryx2
npm install
npm run dev
```

### Agent Project
```bash
cd /Users/abdulrehman/products/mapryx2/mapryx-agent
npm install
npm run dev
```

### Agent Configuration
The agent is configured via:
- `mapryx-agent/wrangler.jsonc` - Worker configuration
- `mapryx-agent/src/server.ts` - Agent logic
- `mapryx-agent/src/app.tsx` - UI components

## Integration Points

### 1. API Endpoint
Agent exposes `/api/chat` endpoint for chat interactions.

### 2. Tool Execution
Tools are executed via `onToolCall` callback in `useMapryxChatAgent`.

### 3. Authentication
Agent uses Supabase session tokens for authentication.

### 4. Database Context
Agent receives database context from `getDatabaseContext()` in main project.

## Testing

### Unit Tests
```bash
npm test
```

### E2E Tests
```bash
npx playwright test
```

### Linting
```bash
npm run lint
```

### Build
```bash
npm run build
```

## Deployment

### Staging
```bash
cd mapryx-agent
npm run deploy
```

### Production
Deploy via Cloudflare Workers with proper secrets:
- `AGENT_ACCESS_PASSWORD`
- `AGENT_SESSION_SECRET`
- `AGENT_AUTH_TOKEN`

## Troubleshooting

### Common Issues

1. **Spatial extension not working**
   - Run `check_system_status` tool
   - Reload page
   - Verify DuckDB-WASM initialization

2. **Table name errors**
   - Use `inspect_layer` to get internal table name
   - Never use user-facing layer names in SQL

3. **SRID errors**
   - Always use strings: `'EPSG:4326'`
   - Never use integers: `4326`

4. **Function signature errors**
   - Use `query_spatial_functions` to verify syntax
   - Check documentation for correct arguments

## Additional Resources

- [DuckDB-WASM Documentation](https://duckdb.org/docs/api/wasm/)
- [deck.gl Documentation](https://deck.gl/)
- [MapLibre GL JS Documentation](https://maplibre.org/maplibre-gl-js/docs/)
- [Cloudflare Agents Documentation](https://developers.cloudflare.com/agents/)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
