package com.hf.sfm.util;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getComboData_2_2_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetComboData() {
        // Given
        BasePara basePara = new BasePara();
        basePara.setSqlpath("testSqlPath");
        basePara.setPaging(true);
        basePara.setStart(0);
        basePara.setLimit(10);
        basePara.setSort("testSort");
        basePara.setDir("testDir");
        basePara.setQueryValue("testQueryValue");
        basePara.setQuerySql("testQuerySql");
        ListRange listRange = new ListRange();
        listRange.setData(new ArrayList<>());
        listRange.setTotalSize(10);
        when(loader.getRange()).thenReturn(listRange);
        // When
        ListRange result = dataSource.getComboData(basePara);
        // Then
        assertNotNull(result);
        assertEquals(listRange, result);
        verify(loader, times(1)).run(basePara);
        verify(loader, times(1)).collectToMap("combo");
    }
}
