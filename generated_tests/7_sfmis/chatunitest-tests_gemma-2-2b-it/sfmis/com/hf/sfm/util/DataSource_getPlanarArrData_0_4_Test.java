package com.hf.sfm.util;

import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getPlanarArrData_0_4_Test {

    @Test
    void testGetPlanarArrData() {
        DataSource dataSource = mock(DataSource.class);
        BasePara basePara = mock(BasePara.class);
        when(dataSource.getPlanarArrData(basePara)).thenReturn("test");
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals("test", result);
    }
}
