package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

class DataSource_getPlanarArrData_0_1_Test {

    @InjectMocks
    private DataSource dataSource;

    @Mock
    private Loader loader;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetPlanarArrData() {
        // Arrange
        BasePara basePara = new BasePara();
        String expectedResult = "Expected Result";
        when(loader.getArrayResults()).thenReturn(expectedResult);
        // Act
        String result = dataSource.getPlanarArrData(basePara);
        // Assert
        verify(loader).run(basePara);
        assertEquals(expectedResult, result);
    }
}
