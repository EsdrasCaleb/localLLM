package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_maturityDateBelow_36_2_Test {

    @Test
    void testMaturityDateBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        String testDate = "2022-12-31";
        subscription.maturityDateBelow(testDate);
        assertEquals(testDate, subscription.maturityDateBelow());
    }
}
