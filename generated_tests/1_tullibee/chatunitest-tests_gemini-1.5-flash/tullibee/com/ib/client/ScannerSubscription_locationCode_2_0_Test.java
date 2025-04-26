package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    @Test
    void testLocationCode() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.locationCode(), "Location code should be null initially");
        subscription.locationCode("US");
        assertEquals("US", subscription.locationCode(), "Location code should be set correctly");
        subscription.locationCode(null);
        assertNull(subscription.locationCode(), "Location code should be set to null correctly");
        subscription.locationCode("");
        assertEquals("", subscription.locationCode(), "Location code should be set to empty string correctly");
    }
}
