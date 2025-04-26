package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    ScannerSubscription mockObject = Mockito.mock(ScannerSubscription.class);

    @Test
    public void testLocationCode() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String expected = "NY";
        Method method = ScannerSubscription.class.getDeclaredMethod("locationCode");
        method.setAccessible(true);
        Mockito.when(mockObject.locationCode()).thenReturn(expected);
        // Act
        String actual = (String) method.invoke(mockObject);
        // Assert
        assertEquals(expected, actual);
    }
}
