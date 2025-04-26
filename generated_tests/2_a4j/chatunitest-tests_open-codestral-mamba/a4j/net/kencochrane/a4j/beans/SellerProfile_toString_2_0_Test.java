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
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        String expectedString = "SellerProfileDetails{...}\n";
        when(sellerProfileDetails.toString()).thenReturn("SellerProfileDetails{...}");
        String actualString = sellerProfile.toString();
        assertEquals(expectedString, actualString);
    }
}
