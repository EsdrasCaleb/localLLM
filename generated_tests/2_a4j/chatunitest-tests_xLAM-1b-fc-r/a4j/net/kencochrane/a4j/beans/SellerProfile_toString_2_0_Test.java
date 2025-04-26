package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SellerProfile_toString_2_0_Test {

    @Test
    void testToString() {
        // Arrange
        SellerProfileDetails mockDetails = Mockito.mock(SellerProfileDetails.class);
        Mockito.when(mockDetails.toString()).thenReturn("Mock Details");
        SellerProfile sellerProfile = new SellerProfile();
        sellerProfile.setSellerProfileDetails(mockDetails);
        // Act
        String result = sellerProfile.toString();
        // Assert
        assertEquals("Mock Details\n", result);
    }
}
