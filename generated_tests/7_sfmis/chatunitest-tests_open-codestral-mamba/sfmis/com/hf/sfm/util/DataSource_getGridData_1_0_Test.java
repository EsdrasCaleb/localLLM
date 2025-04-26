package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
public class DataSource_getGridData_1_0_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetGridData() {
        BasePara basePara = new BasePara();
        // Set up the expected behavior of the loader mock
        when(loader.getRange()).thenReturn(new ListRange());
        ListRange result = dataSource.getGridData(basePara);
        // Verify that the loader's run method was called with the basePara argument
        verify(loader, times(1)).run(basePara);
        // Verify that the loader's collectToMap method was called
        verify(loader, times(1)).collectToMap();
        // Assert that the result is not null (you can add more assertions based on your requirements)
        assertNotNull(result);
    }
}
