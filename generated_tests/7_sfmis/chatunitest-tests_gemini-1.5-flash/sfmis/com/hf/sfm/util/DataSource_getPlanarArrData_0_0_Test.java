package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import javax.servlet.http.HttpSession;

@ExtendWith(MockitoExtension.class)
class DataSource_getPlanarArrData_0_0_Test {

    @Mock
    private Loader loader;

    @InjectMocks
    private DataSource dataSource;

    @Test
    void testGetPlanarArrData_success() {
        BasePara basePara = new BasePara();
        when(loader.getArrayResults(basePara)).thenReturn("Success");
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals("Success", result);
    }

    @Test
    void testGetPlanarArrData_failure() {
        BasePara basePara = new BasePara();
        when(loader.getArrayResults(basePara)).thenReturn("Failure");
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals("Failure", result);
    }

    @Test
    void testGetPlanarArrData_nullResult() {
        BasePara basePara = new BasePara();
        when(loader.getArrayResults(basePara)).thenReturn(null);
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals(null, result);
    }

    @Test
    void testGetPlanarArrData_emptyResult() {
        BasePara basePara = new BasePara();
        when(loader.getArrayResults(basePara)).thenReturn("");
        String result = dataSource.getPlanarArrData(basePara);
        assertEquals("", result);
    }

    static class BasePara {
    }

    static class Loader {

        public void run(BasePara basePara) {
        }

        public String getArrayResults(BasePara basePara) {
            return "Default";
        }
    }

    static class DataSource {

        private final Loader loader;

        DataSource(Loader loader) {
            this.loader = loader;
        }

        public String getPlanarArrData(BasePara basePara) {
            return loader.getArrayResults(basePara);
        }
    }
}
