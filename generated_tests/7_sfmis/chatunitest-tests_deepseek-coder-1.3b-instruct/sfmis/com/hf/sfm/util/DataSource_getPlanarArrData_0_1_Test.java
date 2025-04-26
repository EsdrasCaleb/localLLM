package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
public class DataSource_getPlanarArrData_0_1_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @Test
    public void testGetPlanarArrData() {
        // given
        BasePara basePara = mock(BasePara.class);
        String expectedResult = "expected result";
        when(loader.getArrayResults()).thenReturn(expectedResult);
        // when
        String result = dataSource.getPlanarArrData(basePara);
        // then
        verify(loader, times(1)).run(basePara);
        assertEquals(expectedResult, result);
    }
}
