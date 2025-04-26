package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_scanCode_24_2_Test {

    private String scanCode;

    @Test
    public void testScanCode() throws Exception {
        // <Buggy Line>: cannot find symbol  symbol:   method getScanCode()  location: variable scannerSubscription of type com.ib.client.ScannerSubscription
        assertEquals("12345", this.scanCode);
    }

    public String getScanCode() {
        return this.scanCode;
    }

    public void setScanCode(String scanCode) {
        this.scanCode = scanCode;
    }
}
