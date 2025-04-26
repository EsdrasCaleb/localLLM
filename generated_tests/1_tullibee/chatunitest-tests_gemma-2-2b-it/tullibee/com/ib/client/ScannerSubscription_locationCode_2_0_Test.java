package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    @Test
    void locationCode() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertEquals("locationCode", scannerSubscription.locationCode());
    }
}
