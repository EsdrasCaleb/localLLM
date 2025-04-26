package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SellerProfile_toString_2_0_Test {

    @Mock
    private SellerProfileDetails sellerProfileDetails;

    @InjectMocks
    private SellerProfile sellerProfile;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        // Arrange
        String expectedDetailsString = "Mocked SellerProfileDetails";
        when(sellerProfileDetails.toString()).thenReturn(expectedDetailsString);
        // Act
        String result = sellerProfile.toString();
        // Assert
        assertEquals(expectedDetailsString + "\n", result);
    }
}
