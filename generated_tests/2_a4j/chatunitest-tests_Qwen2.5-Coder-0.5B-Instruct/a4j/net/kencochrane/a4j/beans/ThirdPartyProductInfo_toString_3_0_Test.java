package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ThirdPartyProductInfo_toString_3_0_Test {

    @Test
    public void testToString() {
        // Create a mock instance of ThirdPartyProductInfo
        ThirdPartyProductInfo mockInfo = mock(ThirdPartyProductInfo.class);
        // Arrange: Set up the mock behavior
        // No products
        when(mockInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        // Act: Call the toString method
        String result = mockInfo.toString();
        // Assert: Verify the output
        assertEquals("productOffers is null ", result);
    }
}
