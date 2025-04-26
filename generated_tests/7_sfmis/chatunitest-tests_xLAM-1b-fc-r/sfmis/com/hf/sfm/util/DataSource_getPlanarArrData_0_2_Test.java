package com.hf.sfm.util;

import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

@MockitoSettings(strictness = Strictness.LENIENT)
public class DataSource_getPlanarArrData_0_2_Test {

    @Mock
    private DataSource dataSource;

    @Test
    public void testGetPlanarArrData() {
        BasePara basePara = new BasePara();
        when(dataSource.getPlanarArrData(basePara)).thenReturn("testData");
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals("testData", result);
    }
}
