package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
class DataSource_getComboData_2_0_Test {

    @InjectMocks
    private DataSource dataSource;

    @Mock
    private Loader loader;

    @Mock
    private BasePara basePara;

    @Mock
    private ListRange listRange;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetComboData_WithPaging() {
        // Arrange
        when(basePara.getSqlpath()).thenReturn("test/sqlpath");
        when(basePara.isPaging()).thenReturn(true);
        when(basePara.getStart()).thenReturn(0);
        when(basePara.getLimit()).thenReturn(10);
        when(basePara.getSort()).thenReturn("id");
        when(basePara.getDir()).thenReturn("asc");
        when(basePara.getQueryValue()).thenReturn("value");
        when(basePara.getQuerySql()).thenReturn("SELECT * FROM table");
        when(loader.getRange()).thenReturn(listRange);
        when(listRange.getData()).thenReturn(new ArrayList<>());
        when(listRange.getTotalSize()).thenReturn(0);
        // Act
        ListRange result = dataSource.getComboData(basePara);
        // Assert
        assertNotNull(result);
        assertEquals(0, result.getTotalSize());
        verify(loader).run(basePara);
        verify(loader).collectToMap("combo");
    }

    @Test
    void testGetComboData_WithoutPaging() {
        // Arrange
        when(basePara.getSqlpath()).thenReturn("test/sqlpath");
        when(basePara.isPaging()).thenReturn(false);
        when(basePara.getStart()).thenReturn(-1);
        when(basePara.getLimit()).thenReturn(0);
        when(basePara.getSort()).thenReturn("id");
        when(basePara.getDir()).thenReturn("desc");
        when(basePara.getQueryValue()).thenReturn("value");
        when(basePara.getQuerySql()).thenReturn("SELECT * FROM table");
        when(loader.getRange()).thenReturn(listRange);
        when(listRange.getData()).thenReturn(new ArrayList<>());
        when(listRange.getTotalSize()).thenReturn(0);
        // Act
        ListRange result = dataSource.getComboData(basePara);
        // Assert
        assertNotNull(result);
        assertEquals(0, result.getTotalSize());
        verify(loader).run(basePara);
        verify(loader).collectToMap("combo");
    }
}
