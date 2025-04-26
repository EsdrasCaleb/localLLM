package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class SellerProfile_toString_2_2_Test {

    @Mock
    private SellerProfileDetails sellerProfileDetails;

    @InjectMocks
    private SellerProfile sellerProfile;

    @Test
    public void testToString() {
        // Arrange
        List<String> expectedOutput = new ArrayList<>();
        expectedOutput.add("SellerProfileDetails{id=1,name=John Doe,email=john.doe@example.com,address=123 Main St,Anytown, USA");
        // Act
        String output = sellerProfile.toString();
        // Assert
        assertEquals(expectedOutput, output);
    }
}
