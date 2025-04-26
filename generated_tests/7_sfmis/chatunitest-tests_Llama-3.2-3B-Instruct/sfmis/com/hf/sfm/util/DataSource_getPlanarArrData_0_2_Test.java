package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class DataSource_getPlanarArrData_0_2_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @Test
    public void testGetPlanarArrData() throws Exception {
        // Arrange
        BasePara basePara = new BasePara();
        String expectedArrayResults = "expectedArrayResults";
        // Set up the loader to return a specific result
        when(loader.getArrayResults()).thenReturn(expectedArrayResults);
        // Act
        String actualArrayResults = dataSource.getPlanarArrData(basePara);
        // Assert
        assertEquals(expectedArrayResults, actualArrayResults);
    }

    @Test
    public void testGetPlanarArrData_NullBasePara() {
        // Arrange
        BasePara basePara = null;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> dataSource.getPlanarArrData(basePara));
    }

    @Test
    public void testGetPlanarArrData_LoaderReturnsNull() {
        // Arrange
        BasePara basePara = new BasePara();
        when(loader.getArrayResults()).thenReturn(null);
        // Act
        String actualArrayResults = dataSource.getPlanarArrData(basePara);
        // Assert
        assert actualArrayResults == null;
    }
}
