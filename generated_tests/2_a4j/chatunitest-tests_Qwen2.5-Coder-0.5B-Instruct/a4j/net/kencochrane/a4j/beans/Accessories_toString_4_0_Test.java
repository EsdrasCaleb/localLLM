package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Accessories_toString_4_0_Test {

    @Test
    public void testToString() {
        // Create a mock object for the Accessories class
        Accessories mockAccessories = mock(Accessories.class);
        // Set the expected behavior of the getAccessory method
        when(mockAccessories.getAccessory()).thenReturn(new String[] { "MiniProduct", "Other Product", "Third Product" });
        // Call the toString method on the mock object
        String result = mockAccessories.toString();
        // Verify that the result matches the expected string
        assertEquals("Accessories is null or size 0\nMiniProduct - MiniProduct\nOther Product - Other Product\nThird Product - Third Product", result);
    }
}
