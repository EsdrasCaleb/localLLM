package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_6_0_Test {

    private ScannerSubscription scannerSubscription;

    @Test
    public void testAboveVolume() throws Exception {
        scannerSubscription = new ScannerSubscription();
        Method method = ScannerSubscription.class.getDeclaredMethod("aboveVolume", new Class[] {});
        method.setAccessible(true);
        assertEquals(Integer.MAX_VALUE, method.invoke(scannerSubscription, new Object[] {}));
    }
}
