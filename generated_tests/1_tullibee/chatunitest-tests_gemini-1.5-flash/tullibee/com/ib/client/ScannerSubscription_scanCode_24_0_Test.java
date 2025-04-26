package com.ib.client;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_scanCode_24_0_Test {

    @Test
    void testScanCodeNull() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode(null);
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        assertEquals(null, field.get(scannerSubscription));
    }

    @Test
    void testScanCodeEmpty() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode("");
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        assertEquals("", field.get(scannerSubscription));
    }

    @Test
    void testScanCodeNonEmpty() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String testCode = "12345";
        scannerSubscription.scanCode(testCode);
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        assertEquals(testCode, field.get(scannerSubscription));
    }

    @Test
    void testScanCodeWhitespace() throws Exception {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode("   ");
        Field field = ScannerSubscription.class.getDeclaredField("m_scanCode");
        field.setAccessible(true);
        assertEquals("   ", field.get(scannerSubscription));
    }
}
