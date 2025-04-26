package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @Test
    void moodyRatingBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow("Moody's");
        assertEquals("Moody's", scannerSubscription.moodyRatingBelow());
    }
}
