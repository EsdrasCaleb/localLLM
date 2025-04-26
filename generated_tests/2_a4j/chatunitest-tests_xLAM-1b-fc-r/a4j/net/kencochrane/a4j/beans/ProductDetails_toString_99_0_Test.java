package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.math.BigDecimal;
import java.text.DecimalFormat;

class ProductDetails_toString_99_0_Test {

    @Test
    void getSavings() {
        ProductDetails product = new ProductDetails();
        product.setListPrice("100");
        product.setOurPrice("80");
        String savings = product.getSavings();
        assertNotNull(savings);
    }
}
