package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_0_Test {

    @Test
    public void testToString() {
        // Arrange
        SellerProfile sellerProfile = Mockito.mock(SellerProfile.class);
        SellerProfileDetails sellerProfileDetails = Mockito.mock(SellerProfileDetails.class);
        // Act
        sellerProfile.setSellerProfileDetails(sellerProfileDetails);
        // Assert
        Mockito.when(sellerProfile.getSellerProfileDetails()).thenReturn(sellerProfileDetails);
        Mockito.when(sellerProfileDetails.toString()).thenReturn("SellerProfileDetails");
        // Verify
        assertEquals("SellerProfileDetails", sellerProfile.toString());
    }
}
