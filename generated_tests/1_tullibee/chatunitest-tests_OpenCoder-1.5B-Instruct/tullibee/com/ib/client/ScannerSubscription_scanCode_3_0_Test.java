package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_0_Test {

    @Test
    public void testScanCode() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException, InstantiationException {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        String expected = "SCAN_CODE";
        String methodName = "scanCode";
        Class<?>[] parameterTypes = new Class<?>[] {};
        Object[] args = new Object[] {};
        String result = (String) scannerSubscription.getClass().getMethod(methodName, parameterTypes).invoke(scannerSubscription, args);
        assertEquals(expected, result);
    }
}
