package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_36_0_Test {

    @Test
    public void testMaturityDateBelow() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String inputDate = "2023-12-31";
        scannerSubscription.maturityDateBelow(inputDate);
        Method method = ScannerSubscription.class.getDeclaredMethod("maturityDateBelow", String.class);
        method.setAccessible(true);
        String result = (String) method.invoke(scannerSubscription, inputDate);
        assertEquals(inputDate, result);
    }
}
