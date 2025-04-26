package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testExcludeConvertible() throws Exception {
        scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Method method = ScannerSubscription.class.getDeclaredMethod("excludeConvertible");
        method.setAccessible(true);
        Mockito.when(scannerSubscription.excludeConvertible()).thenReturn("EXCLUDE_CONVERTIBLE");
        String result = (String) method.invoke(scannerSubscription);
        assertEquals("EXCLUDE_CONVERTIBLE", result);
    }
}
