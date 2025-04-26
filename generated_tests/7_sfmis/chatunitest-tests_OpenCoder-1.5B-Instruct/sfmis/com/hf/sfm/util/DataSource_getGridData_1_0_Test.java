package com.hf.sfm.util;

import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpSession;

class DataSource_getGridData_1_0_Test {

    @Test
    void testGetGridData() throws Exception {
        // Arrange
        DataSource dataSource = Mockito.mock(DataSource.class);
        BasePara basePara = Mockito.mock(BasePara.class);
        ListRange listRange = Mockito.mock(ListRange.class);
        Mockito.when(dataSource.getGridData(basePara)).thenReturn(listRange);
        // Act
        Method method = DataSource.class.getDeclaredMethod("getGridData", BasePara.class);
        method.setAccessible(true);
        ListRange result = (ListRange) method.invoke(dataSource, basePara);
        // Assert
        Mockito.verify(dataSource).getGridData(basePara);
        assert result != null;
    }
}
