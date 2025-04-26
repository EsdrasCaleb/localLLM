package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class MiniProduct_toString_12_0_Test {

    @Mock
    private MiniProduct mockMiniProduct;

    @InjectMocks
    private MiniProduct miniProduct;

    @Test
    void testToString() {
        // Arrange
        String expectedString = "asin \n name \n manufacturer \n price \n imageURL";
        // Act
        String actualString = miniProduct.toString();
        // Assert
        assertEquals(expectedString, actualString);
    }
}
