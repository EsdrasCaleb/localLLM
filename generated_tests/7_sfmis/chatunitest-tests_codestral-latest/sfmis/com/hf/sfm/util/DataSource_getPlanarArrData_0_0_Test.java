package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
public class DataSource_getPlanarArrData_0_0_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    private BasePara basePara;

    @BeforeEach
    public void setUp() {
        basePara = new BasePara();
    }

    @Test
    public void testGetPlanarArrData() {
        // Arrange
        String expectedResult = "MockedArrayResults";
        when(loader.getArrayResults()).thenReturn(expectedResult);
        // Act
        String result = dataSource.getPlanarArrData(basePara);
        // Assert
        verify(loader).run(basePara);
        assertEquals(expectedResult, result);
    }
}
