package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateBelow_15_0_Test {

    @Test
    void testMaturityDateBelow_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedMaturityDate = "2024-10-26";
        subscription.maturityDateBelow(expectedMaturityDate);
        String actualMaturityDate = subscription.maturityDateBelow();
        assertEquals(expectedMaturityDate, actualMaturityDate);
    }

    @Test
    void testMaturityDateBelow_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow(null);
        assertNull(subscription.maturityDateBelow());
    }

    @Test
    void testMaturityDateBelow_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow("");
        assertEquals("", subscription.maturityDateBelow());
    }

    @Test
    void testMaturityDateBelow_initialDefault() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Initially null
        assertNull(subscription.maturityDateBelow());
    }
}
