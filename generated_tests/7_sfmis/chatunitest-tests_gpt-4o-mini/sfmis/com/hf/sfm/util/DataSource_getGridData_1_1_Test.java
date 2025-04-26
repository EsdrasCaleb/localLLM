package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
public class DataSource_getGridData_1_1_Test {

    @InjectMocks
    private DataSource dataSource;

    @Mock
    private Loader loader;

    @Mock
    private BasePara basePara;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetGridData_withValidParameters() {
        // Arrange
        when(basePara.getSqlpath()).thenReturn("validSqlPath");
        when(basePara.isPaging()).thenReturn(true);
        when(basePara.getStart()).thenReturn(0);
        when(basePara.getLimit()).thenReturn(10);
        when(basePara.getSort()).thenReturn("id");
        when(basePara.getDir()).thenReturn("asc");
        when(basePara.getQueryValue()).thenReturn("queryValue");
        when(basePara.getQuerySql()).thenReturn("querySql");
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>());
        mockListRange.setTotalSize(0);
        when(loader.getRange()).thenReturn(mockListRange);
        // Act
        ListRange result = dataSource.getGridData(basePara);
        // Assert
        assertNotNull(result);
        assertEquals(0, result.getTotalSize());
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    public void testGetGridData_withEmptyData() {
        // Arrange
        when(basePara.getSqlpath()).thenReturn("emptySqlPath");
        when(basePara.isPaging()).thenReturn(false);
        when(basePara.getStart()).thenReturn(0);
        when(basePara.getLimit()).thenReturn(0);
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>());
        mockListRange.setTotalSize(0);
        when(loader.getRange()).thenReturn(mockListRange);
        // Act
        ListRange result = dataSource.getGridData(basePara);
        // Assert
        assertNotNull(result);
        assertEquals(0, result.getTotalSize());
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    public void testGetGridData_withPagination() {
        // Arrange
        when(basePara.getSqlpath()).thenReturn("paginationSqlPath");
        when(basePara.isPaging()).thenReturn(true);
        when(basePara.getStart()).thenReturn(5);
        when(basePara.getLimit()).thenReturn(10);
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(List.of("data1", "data2")));
        mockListRange.setTotalSize(2);
        when(loader.getRange()).thenReturn(mockListRange);
        // Act
        ListRange result = dataSource.getGridData(basePara);
        // Assert
        assertNotNull(result);
        assertEquals(2, result.getTotalSize());
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }
}
