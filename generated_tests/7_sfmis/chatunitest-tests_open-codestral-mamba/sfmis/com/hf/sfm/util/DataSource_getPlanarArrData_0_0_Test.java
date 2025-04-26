package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

public class DataSource_getPlanarArrData_0_0_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetPlanarArrData() {
        // Assume BasePara has a default constructor
        BasePara basePara = new BasePara();
        String result = dataSource.getPlanarArrData(basePara);
        verify(loader).run(basePara);
        // Replace "expectedResult" with the expected result
        assertEquals("expectedResult", result);
    }
}
