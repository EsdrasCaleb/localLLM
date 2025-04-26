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
public class MiniProduct_toString_12_1_Test {

    @Mock
    private MiniProduct focal;

    @InjectMocks
    private MiniProduct underTest;

    @Test
    public void testToString() {
        List<String> inputs = new ArrayList<>();
        inputs.add("asin1");
        inputs.add("name1");
        inputs.add("manufacturer1");
        inputs.add("price1");
        inputs.add("imageURL1");
        inputs.add("productUrl1");
        when(focal.getAsin()).thenReturn("asin1");
        when(focal.getName()).thenReturn("name1");
        when(focal.getManufacturer()).thenReturn("manufacturer1");
        when(focal.getPrice()).thenReturn("price1");
        when(focal.getImageURL()).thenReturn("imageURL1");
        when(focal.getProductUrl()).thenReturn("productUrl1");
        String expectedOutput = "asin1 \n name1 \n manufacturer1 \n price1 \n imageURL1";
        String actualOutput = underTest.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
