package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getGridData_1_4_Test {

    @Test
    void testGetGridData() {
        DataSource dataSource = mock(DataSource.class);
        BasePara basePara = mock(BasePara.class);
        when(dataSource.getGridData(basePara)).thenReturn(new ListRange());
        ListRange result = dataSource.getGridData(basePara);
        // Assert the result
        verify(dataSource, times(1)).getGridData(basePara);
        // ...
    }
}
