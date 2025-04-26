package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_toString_4_0_Test {

    @Test
    public void testToString() {
        SimilarProducts similarProducts = new SimilarProducts();
        ArrayList<String> simProducts = new ArrayList<>(Arrays.asList("Product1", "Product2"));
        similarProducts.setProduct(simProducts.toArray(new String[0]));
        assertEquals("Product1\nProduct2\n", similarProducts.toString());
    }
}
