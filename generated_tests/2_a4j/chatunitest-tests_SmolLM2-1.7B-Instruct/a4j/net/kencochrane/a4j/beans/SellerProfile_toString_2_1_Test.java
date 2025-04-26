package net.kencochrane.a4j.beans;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@RunWith(MockitoJUnitRunner.class)
public class SellerProfile_toString_2_1_Test {

    @Mock
    private SellerProfileDetails sellerProfileDetails;

    @InjectMocks
    private SellerProfile sellerProfile;

    @Test
    public void testToString_ValidDetails() {
        // Arrange
        sellerProfile.setSellerProfileDetails(sellerProfileDetails);
        // Act
        String expectedString = sellerProfileDetails.toString();
        // Assert
        assertNotNull(sellerProfile.toString());
        assertEquals(expectedString, sellerProfile.toString());
    }

    @Test
    public void testToString_NullDetails() {
        // Arrange
        when(sellerProfileDetails.toString()).thenReturn(null);
        // Act and Assert
        assertNull(sellerProfile.toString());
    }

    @Test
    public void testToString_EmptyDetails() {
        // Arrange
        when(sellerProfileDetails.toString()).thenReturn("");
        // Act and Assert
        assertEquals(sellerProfileDetails.toString(), sellerProfile.toString());
    }

    @Test
    public void testToString_MultipleDetails() {
        // Arrange
        List<String> details = Arrays.asList("Detail1", "Detail2", "Detail3");
        when(sellerProfileDetails.toString()).thenReturn(details.toString());
        // Act
        String expectedString = details.toString();
        // Assert
        assertEquals(expectedString, sellerProfile.toString());
    }
}
