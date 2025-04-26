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
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetPlanarArrData() {
        // Given
        BasePara basePara = mock(BasePara.class);
        String expectedResults = "Mocked Array Results";
        when(loader.getArrayResults()).thenReturn(expectedResults);
        // When
        String actualResults = dataSource.getPlanarArrData(basePara);
        // Then
        verify(loader).run(basePara);
        assertEquals(expectedResults, actualResults);
    }
}
