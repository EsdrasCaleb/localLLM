package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class SellerProfile_toString_2_0_Test {

    @InjectMocks
    private SellerProfile sellerProfile;

    @Test
    public void testToString_SellerProfileDetailsIsNotNull_WhenToStringIsCalled() {
        // Arrange
        SellerProfileDetails sellerProfileDetails = new SellerProfileDetails();
        sellerProfile.setSellerProfileDetails(sellerProfileDetails);
        // Act
        String result = sellerProfile.toString();
        // Assert
        assertEquals(sellerProfileDetails.toString(), result);
    }

    @Test
    public void testToString_SellerProfileDetailsIsNull_WhenToStringIsCalled() {
        // Act
        String result = sellerProfile.toString();
        // Assert
        assertEquals("null", result);
    }
}
