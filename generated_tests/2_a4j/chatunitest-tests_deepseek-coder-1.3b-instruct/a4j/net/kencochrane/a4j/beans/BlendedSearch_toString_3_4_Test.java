package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BlendedSearch_toString_3_4_Test {

    @Test
    void testToString() throws Exception {
        BlendedSearch blendedSearch = new BlendedSearch();
        ArrayList<ProductLine> productLines = new ArrayList<>();
        productLines.add(new ProductLine());
        productLines.add(new ProductLine());
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        String expected = "ProductLine{productId=null, productName=null, productType=null} \n" + "# of productLines = 2 \n";
        Field field = BlendedSearch.class.getDeclaredField("productLines");
        field.setAccessible(true);
        BlendedSearch blendedSearchMock = Mockito.spy(blendedSearch);
        String result = blendedSearchMock.toString();
        assertEquals(expected, result);
    }
}
