package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_locationCode_2_0_Test {

    @Test
    void locationCode_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.locationCode("New York");
        assertEquals("New York", subscription.locationCode());
    }

    @Test
    void locationCode_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.locationCode("");
        assertEquals("", subscription.locationCode());
    }

    @Test
    void locationCode_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.locationCode(null);
        assertNull(subscription.locationCode());
    }

    @Test
    void locationCode_noInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        // No explicit input, relying on default value
        assertNull(subscription.locationCode());
    }
}
