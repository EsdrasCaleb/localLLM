package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_2_Test {

    @Test
    public void testScanCode() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String testScanCode = "TestScanCode";
        scannerSubscription.scanCode(testScanCode);
        assertEquals(testScanCode, scannerSubscription.scanCode());
    }
}
