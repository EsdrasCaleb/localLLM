package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_0_Test {

    private SellerProfile sellerProfile;

    private SellerProfileDetails mockSellerProfileDetails;

    @BeforeEach
    public void setUp() {
        sellerProfile = new SellerProfile();
        mockSellerProfileDetails = Mockito.mock(SellerProfileDetails.class);
    }

    @Test
    public void testToString_withDefaultSellerProfileDetails() {
        // Arrange
        String expectedOutput = sellerProfile.getSellerProfileDetails() + "\n";
        // Act
        String actualOutput = sellerProfile.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_withMockedSellerProfileDetails() {
        // Arrange
        Mockito.when(mockSellerProfileDetails.toString()).thenReturn("Mocked Seller Profile Details");
        sellerProfile.setSellerProfileDetails(mockSellerProfileDetails);
        String expectedOutput = "Mocked Seller Profile Details\n";
        // Act
        String actualOutput = sellerProfile.toString();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
