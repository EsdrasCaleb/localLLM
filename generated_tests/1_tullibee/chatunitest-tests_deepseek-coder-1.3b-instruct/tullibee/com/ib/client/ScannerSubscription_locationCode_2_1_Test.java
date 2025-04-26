package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_1_Test {

    @Test
    public void testLocationCode() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String expectedLocationCode = "Test Location";
        scannerSubscription.locationCode(expectedLocationCode);
        assertEquals(expectedLocationCode, scannerSubscription.locationCode());
    }
}
