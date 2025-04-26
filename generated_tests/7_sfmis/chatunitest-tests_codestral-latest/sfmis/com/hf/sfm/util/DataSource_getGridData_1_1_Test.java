package com.hf.sfm.util;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
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
    private ListRange listRange;

    @BeforeEach
    public void setUp() throws Exception {
        Field loaderField = DataSource.class.getDeclaredField("loader");
        loaderField.setAccessible(true);
        loaderField.set(dataSource, loader);
    }

    @Test
    public void testGetGridData() throws Exception {
        BasePara basePara = new BasePara();
        basePara.setSqlpath("testSqlPath");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        basePara.setSort("testSort");
        basePara.setDir("testDir");
        basePara.setQueryValue("testQueryValue");
        basePara.setQuerySql("testQuerySql");
        when(loader.getRange()).thenReturn(listRange);
        ListRange result = dataSource.getGridData(basePara);
        verify(loader).run(basePara);
        verify(loader).collectToMap();
        assertEquals(listRange, result);
    }
}
