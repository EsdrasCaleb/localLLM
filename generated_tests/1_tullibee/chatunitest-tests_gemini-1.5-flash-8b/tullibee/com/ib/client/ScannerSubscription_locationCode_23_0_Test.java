package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_locationCode_23_0_Test {

    @Test
    void locationCode_setsLocationCode() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid input string
        String locationCode = "TestLocation";
        subscription.locationCode(locationCode);
        Field locationCodeField = ScannerSubscription.class.getDeclaredField("m_locationCode");
        locationCodeField.setAccessible(true);
        assertEquals(locationCode, locationCodeField.get(subscription));
        // Test with null input
        subscription = new ScannerSubscription();
        subscription.locationCode(null);
        locationCodeField.setAccessible(true);
        assertNull(locationCodeField.get(subscription));
        // Test with empty string input
        subscription = new ScannerSubscription();
        subscription.locationCode("");
        locationCodeField.setAccessible(true);
        assertEquals("", locationCodeField.get(subscription));
    }
}
