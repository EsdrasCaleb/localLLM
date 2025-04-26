package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class DataSource_getComboData_2_4_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        ListRange mockListRange = new ListRange();
        // Corrected to ArrayList
        mockListRange.setData(new ArrayList<>());
        mockListRange.setTotalSize(100);
        when(loader.getRange()).thenReturn(mockListRange);
    }

    @Test
    void testGetComboData() throws Exception {
        // Prepare test data
        BasePara basePara = new BasePara();
        basePara.setSqlpath("test.sql");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        basePara.setSort("id");
        basePara.setDir("asc");
        basePara.setQueryValue("testValue");
        basePara.setQuerySql("testSql");
        // Invoke the method under test
        ListRange result = dataSource.getComboData(basePara);
        // Verify interactions with the Loader
        verify(loader, times(1)).run(basePara);
        verify(loader, times(1)).collectToMap("combo");
        // Assert the result
        assertNotNull(result);
        assertNotNull(result.getData());
        assertEquals(100, result.getTotalSize());
    }

    private void setPrivateField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}
