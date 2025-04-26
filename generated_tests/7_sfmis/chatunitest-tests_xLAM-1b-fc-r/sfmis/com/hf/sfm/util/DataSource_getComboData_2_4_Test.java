package com.hf.sfm.util;

import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getComboData_2_4_Test {

    @Mock
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetComboData() {
        // Given
        BasePara basePara = new BasePara();
        basePara.setSqlpath("sqlpath");
        basePara.setPaging(true);
        basePara.setStart(1);
        basePara.setLimit(10);
        basePara.setSort("sort");
        basePara.setDir("dir");
        basePara.setQueryValue("queryValue");
        basePara.setQuerySql("querySql");
        // When
        when(dataSource.getComboData(basePara)).thenReturn(new ListRange());
        // Then
        ListRange result = dataSource.getComboData(basePara);
        assertEquals(new ListRange(), result);
    }
}
