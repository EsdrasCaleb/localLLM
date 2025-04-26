package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_24_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = Mockito.spy(new ScannerSubscription());
    }

    @Test
    public void testScanCode() {
        String testScanCode = "TEST123";
        scannerSubscription.scanCode(testScanCode);
        assertEquals(testScanCode, scannerSubscription.scanCode());
    }
}
