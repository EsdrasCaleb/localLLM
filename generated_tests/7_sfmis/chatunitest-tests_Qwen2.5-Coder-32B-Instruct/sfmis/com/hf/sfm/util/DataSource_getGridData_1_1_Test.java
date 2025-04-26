package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
public class DataSource_getGridData_1_1_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    private BasePara basePara;

    @BeforeEach
    void setUp() {
        basePara = new BasePara();
        basePara.setSqlpath("testSqlPath");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        basePara.setSort("id");
        basePara.setDir("asc");
        basePara.setQueryValue("testQueryValue");
        basePara.setQuerySql("testQuerySql");
        basePara.setQueryparams(new String[] { "param1:value1", "param2:value2" });
    }

    @Test
    void testGetGridDataWithPagingAndParams() throws Exception {
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(Arrays.asList("data1", "data2")));
        mockListRange.setTotalSize(20);
        when(loader.getRange()).thenReturn(mockListRange);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(mockListRange, result);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    void testGetGridDataWithSingleParam() throws Exception {
        basePara.setQueryparams(new String[] { "param1:value1" });
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(Arrays.asList("data1", "data2")));
        mockListRange.setTotalSize(20);
        when(loader.getRange()).thenReturn(mockListRange);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(mockListRange, result);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    void testGetGridDataWithNoParams() throws Exception {
        basePara.setQueryparams(new String[] {});
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(Arrays.asList("data1", "data2")));
        mockListRange.setTotalSize(20);
        when(loader.getRange()).thenReturn(mockListRange);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(mockListRange, result);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    void testGetGridDataWithNullParams() throws Exception {
        basePara.setQueryparams(null);
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(Arrays.asList("data1", "data2")));
        mockListRange.setTotalSize(20);
        when(loader.getRange()).thenReturn(mockListRange);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(mockListRange, result);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    @Test
    void testGetGridDataWithEmptyParams() throws Exception {
        basePara.setQueryparams(new String[] { "" });
        ListRange mockListRange = new ListRange();
        mockListRange.setData(new ArrayList<>(Arrays.asList("data1", "data2")));
        mockListRange.setTotalSize(20);
        when(loader.getRange()).thenReturn(mockListRange);
        ListRange result = dataSource.getGridData(basePara);
        assertEquals(mockListRange, result);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
    }

    private void setPrivateField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}
