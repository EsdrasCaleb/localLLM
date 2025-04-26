package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_2_Test {

    @Test
    public void testMaturityDateBelow() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.maturityDateBelow("2023-12-31");
        Method method = ScannerSubscription.class.getDeclaredMethod("maturityDateBelow");
        method.setAccessible(true);
        String result = (String) method.invoke(scannerSubscription);
        assertEquals("2023-12-31", result);
    }
}
